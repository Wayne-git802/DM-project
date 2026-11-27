# RECALL-MM 修订公告分析

设计师知道产品大类、品名和外观。如果只能先查两个危险，历史召回能帮他选哪两个？找到提示后，再用公告确认该查哪一个部件。

## 阅读顺序

| 入口 | 回答的问题 |
|---|---|
| [01 公告核对、找规律与分组](01_公告核对与字段关联.ipynb) | 用 6,637 份修订公告找出每类常见的危险。再只用品名和外观分组，揭开产品大类和危险标签核对组间差别。 |
| [02 先查哪两个危险](02_危险分类评估.ipynb) | 在同一组五折比较 kNN、Gini 决策树和 XGBoost 等模型，也检查品名、外观和结构各自的作用。最后复核候选规律，并从「跌落」查到部件。 |
| [主工作流](../PROJECT_WORKFLOW_refine.md) | 按「先找规律、再评估提示、最后查部件」阅读，附有 A、B、C、家电子类与 D 的原实验结果。 |

## 一份公告与一个产品系列

同一官方编号的不同款式合为一行。一个产品系列收到多份官方公告时，每份公告各占一行。做五折或留出核对时，同名公告及从原文核实相连的公告一起分配。不同名称的同型号仍可能遗漏，同名只供筛查。Brompton 电助力折叠自行车的两份公告说明为什么要保留各自的日期和失效原因。 [Brompton 21-116](https://www.cpsc.gov/Recalls/2021/Brompton-Bicycle-Recalls-Electric-Folding-Bicycles-Due-to-Fall-and-Injury-Hazards) · [Brompton 22-155](https://www.cpsc.gov/Recalls/2022/Brompton-Bicycle-Recalls-Foldable-Electric-Bicycles-Due-to-Crash-and-Injury-Hazards)

## 三种不同的用法

- **找规律。** 全量 Apriori 先列每类常见的危险和分母。卡方可检查大类与危险整体上是否有关。聚类先按文字相似性分组，再用保留的公告检查各组危险构成。
- **检验能猜中多少。** 五折让每份公告都有一次折外预测。比较时逐步加入设计阶段可用的信息。模型的词表、PCA、参数和规则按各折的训练与内部验证数据拟合。召回原因用于理解失效和核对标签。
- **复核规律并回查部件。** 四折发现的规则用第五折的真实标注核对，模型提示作为辅助比较。原公告确认失效部位。比例反映已收录公告，估计设计本身的事故概率还需要销量、使用暴露和事故数据。

## 数据与复现

| 保留文件 | 用途 |
|---|---|
| [data/dataset_corrected.json](data/dataset_corrected.json) | 两本 Notebook 的共同输入，一份官方公告一行。 |
| [data/reconciliation_summary.json](data/reconciliation_summary.json) | 清洗前后计数、数据来源和哈希。 |
| [data/source_row_audit.jsonl](data/source_row_audit.jsonl) | 逐来源行的官方编号映射、未解决记录和核对证据。按行存储，便于逐条查询。 |
| [scripts/reconcile.py](scripts/reconcile.py) | 用 RECALL-MM 和 CPSC 快照重新生成上面的三份资料。 |
| [scripts/product_features.py](scripts/product_features.py) | 两本 Notebook 共用的产品文字清理、向量化与公告分组。 |
| [scripts/association_rules.py](scripts/association_rules.py) | 全量发现与折内核对共用的 Apriori 规则统计。 |

复跑分析时直接打开两本 Notebook。依赖 NumPy、pandas、SciPy、Matplotlib、scikit-learn、PyTorch、transformers、mlxtend 和 XGBoost，使用本地 MiniLM 权重时设置 `RECALL_MINILM_PATH`。从原始快照复建时按核对汇总准备三个输入文件，在仓库根目录运行。

```sh
python recall_audit/scripts/reconcile.py dataset.json cpsc_api.json cpsc_listing.csv recall_audit/data
```
