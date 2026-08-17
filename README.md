# MovieLens 1M 推荐系统：无泄漏离线评测

# MovieLens 1M Recommender: Leakage-Safe Offline Evaluation

## 中文

这是一个强调**实验可信度**而非模型数量的离线推荐系统项目。当前主实验采用时间顺序留出法和更贴近真实推荐场景的候选集排序，对比贝叶斯流行度基线与稀疏截断 SVD。

> **项目状态：** `run_experiment.py` 是当前可复现的主实验。`Final_Code.ipynb` 和 `Report.pdf` 作为原始课程项目存档保留，其中采用随机切分的历史指标不能与当前实验结果直接比较。

### 项目背景

本项目最初为 Advanced Python 课程期末项目，后续按照数据科学与机器学习作品集标准进行了重构。项目展示了模块化设计、类型标注、Protocol 与 dataclass、命令行接口、Pathlib 文件管理、NumPy/Pandas 向量化、SciPy 稀疏矩阵、确定性随机数控制和单元测试。

### 为什么要进行这次修订

原课程项目使用随机 80/20 切分，并在之后用于测试的数据上间接进行了 SVD 交叉验证调参。此外，原 Top-K 评测只对测试集中已经出现的物品进行排序，因此结果偏乐观。修订后的实验协议解决了这些问题：

1. 按时间戳对全部交互排序，并以 70% / 10% / 20% 划分训练、验证和测试时段；相同时间戳不会跨越边界。
2. 模型选择仅使用训练集和验证集；模型确定后只评估一次测试时段。
3. 对每位符合条件的测试用户，将相关测试电影（评分 ≥ 4）与 100 部该用户从未交互过的电影组成候选集。
4. 同时报告评分准确性、排序质量、目录覆盖率、新颖度、流行度偏差，以及不同用户活跃度分组的表现。

### 可复现实验结果

MovieLens 1M，随机种子 42，全局时间顺序切分，每用户采样 100 个负样本：

| 模型 | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | Coverage@10 | Novelty@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 贝叶斯流行度基线 | 0.9718 | 0.7744 | 0.6232 | 0.2146 | 0.6636 | 0.9535 | 0.1422 | 10.0743 |
| 截断 SVD（50 个隐因子） | 1.0792 | 0.8718 | 0.5276 | 0.1883 | 0.5798 | 0.9376 | **0.4878** | **10.8785** |

以上结果来自 200,041 条测试交互、288,460 个排序候选和 1,762 位符合条件的用户。在这一协议下，流行度基线的准确性更高；SVD 则以部分准确性为代价，显著提高了覆盖率和新颖度。这一准确性—多样性权衡是当前项目能够可靠支持的主要结论。

采样指标依赖候选集构建方式，不能直接与全量排序或使用不同负样本数量的实验比较。完整协议和低/中/高用户活跃度结果见 `results/metrics.json`。

### 复现方式

推荐使用 Python 3.10 或 3.11：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py --models popularity svd --negatives 100
pytest -q
```

输出文件：

- `results/metrics.json`：完整协议、指标、参数和用户活跃度分组结果
- `results/model_comparison.csv`：精简模型对比表

可选的 Optuna 搜索只使用训练集和验证集。选择 SVD 配置后，程序会在训练集与验证集的合集上重新训练，并只进行一次最终测试：

```bash
python run_experiment.py --models svd --n-trials 10 --negatives 100
```

### 仓库结构

| 路径 | 用途 |
|---|---|
| `run_experiment.py` | 可复现的命令行实验入口 |
| `src/data.py` | 带类型定义的 MovieLens 数据加载 |
| `src/evaluation.py` | 时间切分、候选集构建和指标计算 |
| `src/models.py` | 贝叶斯流行度模型和稀疏截断 SVD |
| `tests/` | 切分、负采样和指标逻辑单元测试 |
| `results/` | 机器可读的实验结果 |
| `docs/PROJECT_AUDIT.md` | 项目结论审计和简历使用边界 |
| `Final_Code.ipynb` | 原始课程探索 Notebook（历史存档） |
| `Report.pdf` | 原始课程报告（历史存档） |

### 项目结论边界

- **不声称完成 NMF 对比：** 原 Notebook 仅导入 NMF，未训练或评测。
- **不声称完成 SVD++ 对比：** 最终训练和评测没有保留输出。
- **不声称 Hybrid 改善冷启动：** 原模型实现了自适应权重，但没有严格的新物品或低活跃用户对照实验。
- **不声称 Numba 普遍加速：** 原 Popularity 首次运行包含 JIT 编译，结果仅为 0.14x；另一次 Average Rating 结果为 1.89x。可靠基准应包含预热、多次重复和不确定性。
- **不把旧 SVD 数字作为当前结果：** 原 Notebook 的随机切分结果为 RMSE 0.8612、MAE 0.6756；报告则写为 0.8587、0.6774。二者均未用于当前结果表。

### 数据与许可证

MovieLens 数据由 [GroupLens](https://grouplens.org/datasets/movielens/1m/) 提供，并继续受其使用条款约束。仓库代码采用 [MIT License](LICENSE)。

数据集引用：F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems 5(4). DOI: [10.1145/2827872](https://doi.org/10.1145/2827872)。

---

## English

This offline recommendation project emphasizes **experimental credibility** rather than model count. The primary experiment compares a Bayesian popularity baseline with sparse truncated SVD using a chronological holdout and realistic sampled-candidate ranking.

> **Status:** `run_experiment.py` is the reproducible primary experiment. `Final_Code.ipynb` and `Report.pdf` are retained as legacy course artifacts containing earlier random-split analyses; their metrics are not directly comparable with the current protocol.

### Project Context

This project originated as an Advanced Python course final project and was subsequently refactored into a data-science and machine-learning portfolio project. It demonstrates modular design, type annotations, Protocol and dataclass usage, command-line interfaces, pathlib-based file handling, NumPy/Pandas vectorization, SciPy sparse matrices, deterministic random-state management, and unit testing.

### Why this revision matters

The original course submission used a random 80/20 split and indirectly tuned SVD on data later reused for testing. It also ranked only items already present in the test set, producing optimistic Top-K scores. The revised protocol fixes both issues:

1. Sort all interactions by timestamp and split them 70% / 10% / 20% into train, validation, and test periods; timestamp ties never cross a boundary.
2. Use only train and validation data for model selection; evaluate the test period once after the configuration is fixed.
3. For every eligible test user, rank all relevant test movies (rating ≥ 4) together with 100 seeded movies the user never interacted with.
4. Report rating accuracy, ranking quality, catalog reach, novelty, popularity bias, and performance by pre-test user activity.

### Reproduced results

MovieLens 1M, seed 42, global chronological split, 100 sampled negatives per user:

| Model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | Coverage@10 | Novelty@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bayesian popularity | 0.9718 | 0.7744 | 0.6232 | 0.2146 | 0.6636 | 0.9535 | 0.1422 | 10.0743 |
| Truncated SVD (50 factors) | 1.0792 | 0.8718 | 0.5276 | 0.1883 | 0.5798 | 0.9376 | **0.4878** | **10.8785** |

These values are computed from 200,041 test interactions and 288,460 ranking candidates across 1,762 eligible users. Under this protocol, popularity is the stronger accuracy baseline; SVD trades some accuracy for substantially higher coverage and novelty. This accuracy-diversity trade-off is the primary defensible result.

Sampled metrics depend on candidate construction and are not directly comparable with full-catalog ranking or experiments using a different negative count. See `results/metrics.json` for the full protocol and low/medium/high user-activity slices.

### Reproduce

Python 3.10 or 3.11 is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py --models popularity svd --negatives 100
pytest -q
```

Outputs:

- `results/metrics.json`: complete protocol, metrics, parameters, and activity slices
- `results/model_comparison.csv`: compact model comparison table

Optional Optuna search uses **train/validation only**. The selected SVD configuration is retrained on train + validation before the single final test evaluation:

```bash
python run_experiment.py --models svd --n-trials 10 --negatives 100
```

### Repository layout

| Path | Purpose |
|---|---|
| `run_experiment.py` | Reproducible command-line experiment |
| `src/data.py` | Typed MovieLens data loader |
| `src/evaluation.py` | Temporal split, candidate construction, and metrics |
| `src/models.py` | Bayesian popularity and sparse truncated SVD models |
| `tests/` | Unit tests for split, negative sampling, and metric logic |
| `results/` | Machine-readable reproduced outputs |
| `docs/PROJECT_AUDIT.md` | Claim audit and resume guidance |
| `Final_Code.ipynb` | Legacy exploratory course notebook |
| `Report.pdf` | Legacy course report |

### Claim boundaries

- **NMF is not claimed:** it was imported in the original notebook but never trained or evaluated.
- **SVD++ is not claimed:** final training and evaluation output was not retained.
- **No cold-start improvement is claimed:** the original hybrid used adaptive weights but did not run a controlled new-item or low-activity experiment.
- **No general Numba speedup is claimed:** the original popularity run was 0.14x when JIT compilation was included, while one average-rating run was 1.89x. A defensible benchmark requires warm-up, repeated timings, and uncertainty.
- **Legacy SVD values are not current:** the original notebook retained RMSE 0.8612 and MAE 0.6756 under a random split, while the report listed 0.8587 and 0.6774. Neither is used in the current table.

### Data and license

MovieLens data is provided by [GroupLens](https://grouplens.org/datasets/movielens/1m/) and remains subject to its usage terms. Repository code is released under the [MIT License](LICENSE).

Dataset citation: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems 5(4). DOI: [10.1145/2827872](https://doi.org/10.1145/2827872).

